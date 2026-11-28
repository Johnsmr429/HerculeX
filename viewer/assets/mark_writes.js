function markWrites(scan, page, redraw) {
    let queue = Promise.resolve(),
        pending = 0,
        failed = false;

    function warn(event) {
        event.preventDefault();
        event.returnValue = true;
    }

    async function post(body) {
        const reply = await fetch("/marks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ scan, ...(page ? { page } : {}), ...body }),
        });
        if (!reply.ok) {
            throw new Error("Could not save label");
        }
        return reply.json();
    }

    function enqueue(write) {
        if (++pending === 1) {
            addEventListener("beforeunload", warn);
        }
        redraw(true);
        queue = queue.then(async () => {
            try {
                await write();
            } catch (_) {
                failed = true;
            } finally {
                if (--pending === 0) {
                    removeEventListener("beforeunload", warn);
                }
                redraw(true);
            }
        });
    }

    return {
        status: () => (pending ? " · saving" : "") + (failed ? " · label change not saved" : ""),
        add(mark, layer) {
            const { kind, width_mm, points } = mark;
            enqueue(async () => {
                mark.id = (await post({ kind, width_mm, points, layer })).id;
            });
        },
        remove(mark, discard) {
            mark.removing = true;
            enqueue(async () => {
                try {
                    if (mark.id) {
                        await post({ remove: mark.id });
                    }
                    discard(mark);
                } finally {
                    mark.removing = false;
                }
            });
        },
    };
}
